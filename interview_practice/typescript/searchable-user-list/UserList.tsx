import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";

import { fetchUsers } from "./userData";
import { filterUsers } from "./filterUsers";
import type { User, UserListState } from "./types";

type UserListProps = {
  title?: string;
};

const initialState: UserListState = {
  users: [],
  searchTerm: "",
  isLoading: true,
  error: null,
};

export function UserList({ title = "User Directory" }: UserListProps) {
  const [{ users, searchTerm, isLoading, error }, setState] =
    useState<UserListState>(initialState);


  useEffect(() => {
    let isMounted = true;
    async function loadUsers() {
      try {
        const data = await fetchUsers();

        if (!isMounted) {
          return;
        }

        setState((current) => ({
          ...current,
          users: data,
          isLoading: false,
        }));
      } catch {
        if (!isMounted) {
          return;
        }

        setState((current) => ({
          ...current,
          error: "Failed to load users.",
          isLoading: false,
        }));
      }
    }

    void loadUsers();
    return () => {
      isMounted = false;
    };
  }, []);

  const filteredUsers: User[] = filterUsers(users, searchTerm);

  function handleSearchChange(event: ChangeEvent<HTMLInputElement>) {
    const nextValue = event.target.value;

    setState((current) => ({
      ...current,
      searchTerm: nextValue,
    }));
  }

  return (
    <section>
      <h1>{title}</h1>

      <label htmlFor="user-search">Search users</label>
      <input
        id="user-search"
        type="text"
        value={searchTerm}
        onChange={handleSearchChange}
        placeholder="Search by name, email, or company"
      />

      {isLoading ? <p>Loading users...</p> : null}
      {error ? <p role="alert">{error}</p> : null}

      {!isLoading && !error && filteredUsers.length === 0 ? (
        <p>No users found.</p>
      ) : null}

      {!isLoading && !error && filteredUsers.length > 0 ? (
        <ul>
          {filteredUsers.map((user) => (
            <li key={user.id}>
              <strong>{user.name}</strong>
              <div>{user.email}</div>
              <div>{user.company}</div>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
